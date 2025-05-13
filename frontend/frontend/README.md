# AuditPulse Frontend

This is the frontend application for AuditPulse, an AI-powered financial transaction monitoring system.

## Getting Started

### Prerequisites

- Node.js (v18 or later)
- npm (v8 or later)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/auditpulse.git
cd auditpulse/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file:
```bash
cp .env.example .env
```

4. Update the `.env` file with your configuration values.

### Development

Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`.

### Building for Production

Create a production build:
```bash
npm run build
```

The build artifacts will be stored in the `build/` directory.

### Testing

Run the test suite:
```bash
npm test
```

Run the test suite with coverage:
```bash
npm run test:coverage
```

### Linting

Run the linter:
```bash
npm run lint
```

Fix linting issues:
```bash
npm run lint:fix
```

## Project Structure

```
frontend/
├── public/              # Static files
├── src/                 # Source code
│   ├── components/      # React components
│   ├── hooks/          # Custom React hooks
│   ├── pages/          # Page components
│   ├── services/       # API services
│   ├── store/          # State management
│   ├── styles/         # Global styles
│   ├── types/          # TypeScript types
│   └── utils/          # Utility functions
├── .env.example        # Example environment variables
├── .gitignore          # Git ignore file
├── package.json        # Project dependencies
├── tsconfig.json       # TypeScript configuration
└── README.md           # Project documentation
```

## Contributing

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git commit -m "Add your feature"
```

3. Push to your branch:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 